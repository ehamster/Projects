package com.parkingauthorityticketingsystem;


import android.app.Activity;
import android.app.AlertDialog;
import android.app.Fragment;
import android.content.Context;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.support.annotation.Nullable;
import android.support.v4.widget.SwipeRefreshLayout;
import android.support.v7.widget.DividerItemDecoration;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.helper.ItemTouchHelper;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.AutoCompleteTextView;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Toast;

import com.parkingauthorityticketingsystem.database.ShiftlogDBHelper;
import com.parkingauthorityticketingsystem.entity.Shiftlog;
import com.parkingauthorityticketingsystem.location_related.PlacesAutoCompleteAdapter;
import com.parkingauthorityticketingsystem.utils.AbbreviationUtils;
import com.parkingauthorityticketingsystem.utils.WarningDialogUtils;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;




/**
 * A simple {@link Fragment} subclass.
 */
public class CreateShiftlogFragment extends Fragment {
    private ShiftlogDBHelper helper;
    private RecyclerView recyclerView;
    private List<Shiftlog> shiftlogs = new ArrayList<>();
    private RecentEntriesAdapter adapter;
    private Toast toast;
    private Paint p = new Paint();
    private String location;
    double x_coordinate;
    double y_coordinate;
    View view;

    RequestLocInterface requestLocInterface;
    String displayed;
    AutoCompleteTextView autocompleteView;

    private static String TAG = MainActivity.class.getSimpleName();

    private PlacesAutoCompleteAdapter mAdapter;

    HandlerThread mHandlerThread;
    Handler mThreadHandler;

    AlertDialog.Builder builder;

    public CreateShiftlogFragment(){

        if (mThreadHandler == null) {
            // Initialize and start the HandlerThread
            // which is basically a Thread with a Looper
            // attached (hence a MessageQueue)
            mHandlerThread = new HandlerThread(TAG, android.os.Process.THREAD_PRIORITY_BACKGROUND);
            mHandlerThread.start();

            // Initialize the Handler
            mThreadHandler = new Handler(mHandlerThread.getLooper()) {
                @Override
                public void handleMessage(Message msg) {
                    if (msg.what == 1) {
                        ArrayList<String> results = mAdapter.resultList;

                        if (results != null && results.size() > 0) {
                            System.out.println("1Thread: "+Thread.currentThread().getName());
                            try{
                                mAdapter.notifyDataSetChanged();}catch(Exception e){
                                System.out.println("Attention!!!");
                                e.printStackTrace();
                            }
                        }
                        else {
                            mAdapter.notifyDataSetInvalidated();
                        }
                    }
                }
            };
        }
    }


    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        if((++MainActivity.justOnce)==1) {
            view = inflater.inflate(R.layout.fragment_create_shiftlog,container,false);
            MainActivity.Viewholder.setMyView(view);}
        view = MainActivity.Viewholder.getMyView();
        helper = new ShiftlogDBHelper(getContext()); // Create Database with name : shiftlog.db
        recyclerView = (RecyclerView)view.findViewById(R.id.recyclerView);
        recyclerView.setLayoutManager(new LinearLayoutManager(view.getContext()));
        shiftlogs = helper.queryAll(); // Read data from database
        adapter = new RecentEntriesAdapter(shiftlogs);
        recyclerView.setAdapter(adapter);
        recyclerView.addItemDecoration(new DividerItemDecoration(view.getContext(),DividerItemDecoration.VERTICAL));
        //Refresh RecyclerView
        final SwipeRefreshLayout swipeRefreshLayout = (SwipeRefreshLayout) view.findViewById(R.id.refresh);
        swipeRefreshLayout.setColorSchemeColors(Color.argb(150,164,211,238));
        swipeRefreshLayout.setOnRefreshListener(new SwipeRefreshLayout.OnRefreshListener(){

            @Override
            public void onRefresh() {
                swipeRefreshLayout.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        swipeRefreshLayout.setRefreshing(false);

                    }
                },500);
                adapter.refresh(shiftlogs);

            }
        });
        initSwipe();
        view.findViewById(R.id.viewshiftlog_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MapViewFragment mapViewFragment = new MapViewFragment();
                Bundle bundle = new Bundle();  //Save current location
                requestLocInterface.requestLocation("saveLoc");
                String []xy = location.split(" ");
                x_coordinate = Double.parseDouble(xy[0]);
                y_coordinate = Double.parseDouble(xy[1]);
                bundle.putDouble("x",x_coordinate);
                bundle.putDouble("y",y_coordinate);
                mapViewFragment.setArguments(bundle);  // transform location from here to mapView
                getActivity().getFragmentManager().beginTransaction().replace(R.id.fragmentContainer,mapViewFragment)
                        .addToBackStack(null).commit();
                Log.i("aa","aa");
            }
        });
        view.findViewById(R.id.clear_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ((EditText)getActivity().findViewById(R.id.plateNum_et)).setText("");
                ((EditText)getActivity().findViewById(R.id.max_time_et)).setText("");
                ((EditText)getActivity().findViewById(R.id.meterNum_et)).setText("");
                ((EditText)getActivity().findViewById(R.id.streetNum_et)).setText("");
            }

        });
        view.findViewById(R.id.save_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Shiftlog sl = getCurrentInput(view);

                if(sl == null){

                }
                else{
                    requestLocInterface.requestLocation("saveLoc");
                    SimpleDateFormat formatter = new SimpleDateFormat ("MM/dd/yyyy HH:mm:ss");
                    Date curDate = new Date(System.currentTimeMillis());
                    sl.setFirstObserve(formatter.format(curDate));
                    String plate = sl.getPlateNumber();
                    sl.setPlateNumber(plate.toUpperCase());// Change PlateNum to UpperCase
                    sl.setState(AbbreviationUtils.getAbbreviation(sl.getState()));
                    //Use current location to set shiftlog x y coordination.
                    //location = "40 -70";
                    String []xy = location.split(" ");
                    x_coordinate = Double.parseDouble(xy[0]);
                    y_coordinate = Double.parseDouble(xy[1]);
                    System.out.println("This is a check, location:"+location+" x="+ x_coordinate+ " y="+y_coordinate);
                    sl.setX_coordinate(x_coordinate);
                    sl.setY_coordinate(y_coordinate);
                    helper.insertRecord(sl); //Add new item into DB

                    if (toast != null)
                    {
                        toast.setText("Saved Successfully!");
                        toast.setDuration(Toast.LENGTH_SHORT);
                        toast.show();
                    } else
                    {
                        toast = Toast.makeText(getContext(), "Saved Successfully!",Toast.LENGTH_SHORT);
                        toast.show();
                    }
                    shiftlogs =  helper.queryAll();   // Read all items from DB
                    adapter.refresh(shiftlogs);
                }
            }
        });

//        view.findViewById(R.id.listView_btn).setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                SL_ListViewFragment listViewFragment = new SL_ListViewFragment();
//                getActivity().getFragmentManager().beginTransaction().replace(R.id.fragmentContainer,listViewFragment)
//                        .addToBackStack(null).commit();
//
//            }
//        });

        return view;

    }


    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {

        //Autocomplete
        autocompleteView = (AutoCompleteTextView)view.findViewById(R.id.street_name);

        mAdapter = new PlacesAutoCompleteAdapter(getActivity(), R.layout.autocomplete_list_item);
        autocompleteView.setAdapter(mAdapter);

        //  autocompleteView.setAdapter(new PlacesAutoCompleteAdapter(this, R.layout.autocomplete_list_item));

        System.out.println("Thread: "+Thread.currentThread().getName());

        autocompleteView.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {

            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                System.out.println("Just so you know: "+s);
                final String value = s.toString();

                // Remove all callbacks and messages
                mThreadHandler.removeCallbacksAndMessages(null);

                // Now add a new one
                mThreadHandler.postDelayed(new Runnable() {

                    @Override
                    public void run() {
                        // Background thread
                        mAdapter.resultList = mAdapter.mPlaceAPI.autocomplete(value);

                        // Footer
                        if (mAdapter.resultList.size() > 0)
                            mAdapter.resultList.add("footer");

                        // Post to Main Thread
                        mThreadHandler.sendEmptyMessage(1);
                    }
                }, 500);
            }

            @Override
            public void afterTextChanged(Editable s) {
                // doAfterTextChanged();
            }
        });

        autocompleteView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                autocompleteView.setText("");
                System.out.println("CLEARED!");
            }

        });

        //old
        autocompleteView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                // Get data associated with the specified position
                // in the list (AdapterView)
                String description = (String) parent.getItemAtPosition(position);
                String[] s = description.split(",");
                displayed = s[0];
                System.out.println("This too: "+displayed);
                autocompleteView.setText(displayed);
                autocompleteView.setSelection(displayed.length());
            }
        });

    }


    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        Activity activity;
        if(context instanceof Activity){
            activity = (Activity)context;

            // This makes sure that the container activity has implemented
            // the callback interface. If not, it throws an exception
            try {
                requestLocInterface = (RequestLocInterface) activity;
            } catch (ClassCastException e) {
                throw new ClassCastException(activity.toString()
                        + " must implement RequestLocationInterface");
            }
        }
    }

    @Override
    public void onDetach() {
        requestLocInterface = null; // => avoid leaking, thanks @Deepscorn
        super.onDetach();
    }


    public void getCoordinates(String info){
        String[] result = info.split(":");
       // System.out.println("getCoordinates: "+result[1]);
        location = result[1];
    }

    private void initSwipe(){
        ItemTouchHelper.SimpleCallback simpleItemTouchCallback = new ItemTouchHelper.SimpleCallback(0, ItemTouchHelper.LEFT ) {

            @Override
            public boolean onMove(RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder, RecyclerView.ViewHolder target) {

                return false;

            }

            @Override
            public void onSwiped(RecyclerView.ViewHolder viewHolder, int direction) {
                final int position = viewHolder.getAdapterPosition();

                if (direction == ItemTouchHelper.LEFT){
                    new AlertDialog.Builder(getContext())
                            .setTitle("Warning").setMessage("Are you sure to delete ?").setNegativeButton("Confirm", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            Shiftlog sl = adapter.removeItem(position);
                            shiftlogs.remove(sl);
                            helper.deleteRecord(sl.getId());
                        }
                    }).setPositiveButton("Cancel", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                           // adapter.refresh(shiftlogs);
                        }
                    }).create().show();
                    adapter.refresh(shiftlogs);

                }
            }


            @Override
            public void onChildDraw(Canvas c, RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder, float dX, float dY, int actionState, boolean isCurrentlyActive) {

                Bitmap icon;
                if(actionState == ItemTouchHelper.ACTION_STATE_SWIPE){

                    View itemView = viewHolder.itemView;
                    float height = (float) itemView.getBottom() - (float) itemView.getTop();
                    float width = height / 3;

                    if(dX > 0){
                        /*p.setColor(Color.parseColor("#388E3C"));
                        RectF background = new RectF((float) itemView.getLeft(), (float) itemView.getTop(), dX,(float) itemView.getBottom());
                        c.drawRect(background,p);
                        icon = BitmapFactory.decodeResource(getResources(), R.drawable.ic_edit_white);
                        RectF icon_dest = new RectF((float) itemView.getLeft() + width ,(float) itemView.getTop() + width,(float) itemView.getLeft()+ 2*width,(float)itemView.getBottom() - width);
                        c.drawBitmap(icon,null,icon_dest,p);*/
                    } else {
                        p.setColor(Color.parseColor("#FF4500"));
                        RectF background = new RectF((float) itemView.getRight() + dX, (float) itemView.getTop(),(float) itemView.getRight(), (float) itemView.getBottom());
                        c.drawRect(background,p);
                        icon = BitmapFactory.decodeResource(getResources(), R.drawable.ic_delete_white_24dp);
                        RectF icon_dest = new RectF((float) itemView.getRight() - 2*width ,(float) itemView.getTop() + width,(float) itemView.getRight() - width,(float)itemView.getBottom() - width);
                        c.drawBitmap(icon,null,icon_dest,p);
                    }

                }
                super.onChildDraw(c, recyclerView, viewHolder, dX, dY, actionState, isCurrentlyActive);
            }
        };
        ItemTouchHelper itemTouchHelper = new ItemTouchHelper(simpleItemTouchCallback);
        itemTouchHelper.attachToRecyclerView(recyclerView);
    }


    private Shiftlog getCurrentInput(View view) {  //use a Shiftlog object to get current input info
        Shiftlog sl = new Shiftlog();
        if(TextUtils.isEmpty(((EditText)view.findViewById(R.id.plateNum_et)).getText().toString())){
            WarningDialogUtils.warning(getContext(),"PlateNumber_null");
            return null;
        }
        else if(TextUtils.isEmpty(((EditText)view.findViewById(R.id.max_time_et)).getText().toString())) {
            WarningDialogUtils.warning(getContext(),"MaxTime_null");
            return null;
        }
        else {
            sl.setPlateNumber(((EditText) view.findViewById(R.id.plateNum_et)).getText().toString());
            sl.setState(((Spinner) view.findViewById(R.id.state_spinner)).getSelectedItem().toString());
            sl.setMaxTime(Integer.parseInt(((EditText) view.findViewById(R.id.max_time_et)).getText().toString()));
            sl.setMeterNum(((EditText) view.findViewById(R.id.meterNum_et)).getText().toString());
            sl.setStreetNum(((EditText) view.findViewById(R.id.streetNum_et)).getText().toString());
           // sl.setStreet(((EditText) view.findViewById(R.id.street_name)).getText().toString());
            sl.setStreet(autocompleteView.getText().toString());
            sl.setFirstObserve("");
            return sl;
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();

        // Get rid of our Place API Handlers
        if (mThreadHandler != null) {
            mThreadHandler.removeCallbacksAndMessages(null);
            mHandlerThread.quit();
        }
    }
}
